use anyhow::Result;
use jope_learning::promote::*;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier, Mutex,
    },
    thread,
};

#[derive(Default)]
struct State {
    pointers: BTreeMap<PointerKind, VersionedPointer>,
    events: Vec<PublicationEvent>,
}

#[derive(Clone)]
struct MemoryStore {
    state: Arc<Mutex<State>>,
    barrier: Option<Arc<(Barrier, AtomicUsize)>>,
}

impl MemoryStore {
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State::default())),
            barrier: None,
        }
    }

    fn concurrent_read_barrier(&mut self) {
        self.barrier = Some(Arc::new((Barrier::new(2), AtomicUsize::new(2))));
    }

    fn install(&self, kind: PointerKind, pointer: RootPointer) {
        self.state.lock().unwrap().pointers.insert(
            kind,
            VersionedPointer {
                pointer,
                store_version: "v1".into(),
            },
        );
    }

    fn pointer(&self, kind: PointerKind) -> Option<RootPointer> {
        self.state
            .lock()
            .unwrap()
            .pointers
            .get(&kind)
            .map(|v| v.pointer.clone())
    }

    fn events(&self) -> Vec<PublicationEvent> {
        self.state.lock().unwrap().events.clone()
    }
}

impl PublicationStore for MemoryStore {
    fn read_pointer(&self, kind: PointerKind) -> Result<Option<VersionedPointer>> {
        let value = self.state.lock().unwrap().pointers.get(&kind).cloned();
        if let Some(barrier) = &self.barrier {
            if barrier
                .1
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                    (n > 0).then(|| n - 1)
                })
                .is_ok()
            {
                barrier.0.wait();
            }
        }
        Ok(value)
    }

    fn append_event(&self, event: PublicationEvent) -> Result<()> {
        self.state.lock().unwrap().events.push(event);
        Ok(())
    }

    fn compare_exchange(
        &self,
        kind: PointerKind,
        expected_store_version: Option<&str>,
        next: RootPointer,
    ) -> Result<bool> {
        let mut state = self.state.lock().unwrap();
        let actual = state.pointers.get(&kind).map(|v| v.store_version.as_str());
        if actual != expected_store_version {
            return Ok(false);
        }
        let version = actual
            .and_then(|v| v.strip_prefix('v'))
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0)
            + 1;
        state.pointers.insert(
            kind,
            VersionedPointer {
                pointer: next,
                store_version: format!("v{version}"),
            },
        );
        Ok(true)
    }
}

fn d(label: &str) -> String {
    jope_learning::canon::sha256_hex(label.as_bytes())
}
fn rollback(generation: u64, root: &str) -> RollbackRoot {
    RollbackRoot {
        generation,
        root_sha256: d(root),
        closure_sha256: d("rollback-closure"),
        required_objects: [d("rollback-object")].into_iter().collect(),
    }
}
fn initial_root(generation: u64, label: &str) -> RootPointer {
    RootPointer {
        schema: PUBLICATION_SCHEMA.into(),
        generation,
        root_sha256: d(label),
        parent_root_sha256: None,
        decision_sha256: d("initial-decision"),
        closure_sha256: d("initial-closure"),
        rollback: rollback(generation.saturating_sub(1), "older"),
    }
}
fn closure(closure_label: &str, required_labels: &[&str]) -> ClosureProof {
    let required_objects: std::collections::BTreeSet<_> =
        required_labels.iter().map(|s| d(s)).collect();
    ClosureProof {
        closure_sha256: d(closure_label),
        remote_objects: required_objects.clone(),
        required_objects,
        retired_strong_refs: Default::default(),
    }
}
fn grant(purpose: GrantPurpose, generation: u64, parent: Option<String>) -> PublisherGrant {
    PublisherGrant {
        verified_grant_sha256: d("authority-grant"),
        purpose,
        expected_generation: generation,
        expected_parent_sha256: parent,
        expires_at_ms: 1000,
    }
}
fn research_decision(generation: u64, parent: &str, candidate: &str) -> ResearchParentDecision {
    let mut proof = closure(
        "candidate-closure",
        &["candidate-object", "rollback-object"],
    );
    proof.closure_sha256 = d("candidate-closure");
    ResearchParentDecision {
        expected_generation: generation,
        expected_parent_sha256: Some(d(parent)),
        candidate_root_sha256: d(candidate),
        candidate_evidence_sha256: d("candidate-evidence"),
        assessment: Assessment::Qualified,
        compatible_closure_sha256: d("candidate-closure"),
        closure: proof,
        rollback: rollback(generation, parent),
    }
}

#[test]
fn two_cas_contenders_have_one_winner_and_stale_grant_cannot_advance_again() {
    let mut store = MemoryStore::new();
    store.install(
        PointerKind::ResearchParent,
        initial_root(4, "accepted-parent"),
    );
    store.concurrent_read_barrier();
    let root = Arc::new(store);
    let mut contenders = Vec::new();
    for candidate in ["candidate-a", "candidate-b"] {
        let store = root.clone();
        contenders.push(thread::spawn(move || {
            let mut decision = research_decision(4, "accepted-parent", candidate);
            decision.candidate_evidence_sha256 = d(candidate);
            let grant = grant(
                GrantPurpose::ResearchParentAdoption,
                4,
                Some(d("accepted-parent")),
            );
            adopt_research_parent(store.as_ref(), &grant, decision, 50)
        }));
    }
    let results: Vec<_> = contenders.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    let after_race = root.pointer(PointerKind::ResearchParent).unwrap();
    assert_eq!(after_race.generation, 5);
    let stale_grant = grant(
        GrantPurpose::ResearchParentAdoption,
        4,
        Some(d("accepted-parent")),
    );
    assert!(adopt_research_parent(
        root.as_ref(),
        &stale_grant,
        research_decision(4, "accepted-parent", "late"),
        50
    )
    .is_err());
    assert_eq!(
        root.pointer(PointerKind::ResearchParent).unwrap(),
        after_race
    );
    assert_eq!(root.events().len(), 3); // Both contenders and stale candidate evidence remain recorded.
}

#[test]
fn absent_remote_closure_and_retired_strong_refs_block_promotion() {
    let store = MemoryStore::new();
    store.install(PointerKind::ResearchParent, initial_root(2, "parent"));
    let g = grant(GrantPurpose::ResearchParentAdoption, 2, Some(d("parent")));
    let mut missing = research_decision(2, "parent", "candidate");
    missing.closure.remote_objects.clear();
    assert!(adopt_research_parent(&store, &g, missing, 50).is_err());
    let mut retired = research_decision(2, "parent", "candidate-2");
    retired
        .closure
        .retired_strong_refs
        .insert(d("candidate-object"));
    assert!(adopt_research_parent(&store, &g, retired, 50).is_err());
    assert_eq!(
        store
            .pointer(PointerKind::ResearchParent)
            .unwrap()
            .root_sha256,
        d("parent")
    );
    assert_eq!(store.events().len(), 2); // Failed candidates remain immutable evidence events.
}

#[test]
fn negative_or_inconclusive_evidence_retains_parent_and_is_recorded() {
    let store = MemoryStore::new();
    store.install(PointerKind::ResearchParent, initial_root(2, "parent"));
    let g = grant(GrantPurpose::ResearchParentAdoption, 2, Some(d("parent")));
    for assessment in [Assessment::Negative, Assessment::Inconclusive] {
        let mut decision = research_decision(2, "parent", "candidate");
        decision.assessment = assessment;
        let result = adopt_research_parent(&store, &g, decision, 50).unwrap();
        assert!(
            matches!(result, AdoptionResult::Retained { assessment: found, .. } if found == assessment)
        );
        assert_eq!(
            store
                .pointer(PointerKind::ResearchParent)
                .unwrap()
                .root_sha256,
            d("parent")
        );
    }
    assert_eq!(store.events().len(), 2);
}

#[test]
fn rollback_moves_pointer_and_preserves_committed_child_and_ledger_events() {
    let store = MemoryStore::new();
    let research = initial_root(3, "research-parent");
    store.install(PointerKind::ResearchParent, research.clone());
    store.install(PointerKind::Serving, initial_root(6, "serving-parent"));
    record_checkpoint_commit(
        &store,
        CheckpointCommit {
            checkpoint_sha256: d("candidate"),
            candidate_evidence_sha256: d("evidence"),
            closure_sha256: d("serving-closure"),
        },
    )
    .unwrap();
    let serving_decision = ServingPromotionDecision {
        expected_generation: 6,
        expected_parent_sha256: Some(d("serving-parent")),
        expected_research_parent_sha256: research.root_sha256,
        candidate_root_sha256: d("candidate"),
        candidate_evidence_sha256: d("evidence"),
        qualification_sha256: Some(d("independent-qualification")),
        assessment: Assessment::Qualified,
        compatible_closure_sha256: d("serving-closure"),
        closure: closure("serving-closure", &["candidate-object", "rollback-object"]),
        rollback: rollback(6, "serving-parent"),
    };
    let promotion_grant = grant(GrantPurpose::ServingPromotion, 6, Some(d("serving-parent")));
    assert!(matches!(
        promote_serving(&store, &promotion_grant, serving_decision, 50).unwrap(),
        AdoptionResult::Advanced(_)
    ));
    let current = store.pointer(PointerKind::Serving).unwrap();
    assert_eq!(current.root_sha256, d("candidate"));
    let rollback_target = rollback(6, "serving-parent");
    let mut rollback_closure = closure("rollback-closure", &["rollback-object"]);
    rollback_closure.closure_sha256 = rollback_target.closure_sha256.clone();
    let rollback_grant = grant(
        GrantPurpose::Rollback,
        current.generation,
        Some(current.root_sha256.clone()),
    );
    let restored = rollback_root(
        &store,
        &rollback_grant,
        PointerKind::Serving,
        rollback_target,
        rollback_closure,
        d("rollback-evidence"),
        60,
    )
    .unwrap();
    assert_eq!(restored.generation, 8);
    assert_eq!(restored.root_sha256, d("serving-parent"));
    let events = store.events();
    assert!(events.iter().any(|e| matches!(e, PublicationEvent::CheckpointCommitted { checkpoint } if checkpoint.checkpoint_sha256 == d("candidate"))));
    assert!(events
        .iter()
        .any(|e| matches!(e, PublicationEvent::ServingAssessed { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, PublicationEvent::RootRolledBack { .. })));
}
