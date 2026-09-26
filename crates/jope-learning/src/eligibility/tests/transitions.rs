//! W3-T05 at the real ingress: eligibility, HWM initialization, ties and committed
//! improvements are distinct outcomes. Synthetic fixture; not a live search.
use super::*;

fn transitions(state: &State) -> Vec<serde_json::Value> {
    Ledger::open(state.root(), "hwm")
        .events()
        .unwrap()
        .into_iter()
        .map(|event| event["transition"].clone())
        .collect()
}

#[test]
fn eligibility_alone_and_redelivery_never_count_as_improvement() {
    let f = fixture(Task::Regression, 5);
    qualify(&f.state, &f.record, &f.evidence).unwrap();
    // Eligible, yet no HWM transition exists until ingestion commits one.
    assert!(transitions(&f.state).is_empty());
    assert!(f.state.pointers().unwrap().is_empty());
    let candidate = || Candidate {
        record: f.record.clone(),
        blob: b"synthetic candidate".to_vec(),
        chain_from_source: false,
    };
    let first = hwm::ingest(&f.state, None, vec![candidate()], json!({"kind":"w3-t05"})).unwrap();
    let replay = hwm::ingest(&f.state, None, vec![candidate()], json!({"kind":"w3-t05"})).unwrap();
    assert_eq!(
        transitions(&f.state),
        vec![json!("initialized"), json!("tie")]
    );
    assert_eq!(first.transitions.get("initialized"), Some(&1));
    assert_eq!(replay.transitions.get("tie"), Some(&1));
    assert!(
        !first.transitions.contains_key("improved") && !replay.transitions.contains_key("improved")
    );
    let verified = hwm::verify(&f.state, &[]).unwrap();
    assert_eq!((verified.initialized, verified.improved), (1, 0));
    let current = f
        .state
        .read_pointer(&PointerKey::of(&f.record))
        .unwrap()
        .unwrap();
    assert_eq!(current.improvement_number, 0);
    assert_eq!(freeze_teaching(&f.state).unwrap().entries.len(), 1);
}
