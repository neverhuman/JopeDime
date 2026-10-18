use jope_records::research::{
    brief, do_not_repeat, pool_descriptive, state_check, BeliefOutcome, BeliefRecord, BriefEntry,
    CiEvidence, CiStatus, CurrentState, DoNotRepeat, Exclusion, PoolIdentity, PriorRun,
    RepeatSignature, ReplicationOverride, SelectionMode, SourceEvidence, SourceStatus,
    STATE_BYTE_LIMIT,
};

const HEAD: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OLD_HEAD: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn source() -> SourceEvidence {
    SourceEvidence {
        status: SourceStatus::Committed,
        head: Some(HEAD.into()),
    }
}

fn ci(status: CiStatus, head: Option<&str>, run: Option<&str>) -> CiEvidence {
    CiEvidence {
        status,
        head: head.map(str::to_string),
        run_id: run.map(str::to_string),
    }
}

fn signature() -> RepeatSignature {
    RepeatSignature {
        hypothesis: "H-replication".into(),
        endpoint: "held-r2".into(),
        context: "panel-v3".into(),
        support: "dataset-family-2".into(),
        cost: "cpu-2h".into(),
    }
}

fn belief(id: &str, group: &str, outcome: BeliefOutcome) -> BeliefRecord {
    BeliefRecord {
        record_id: id.into(),
        hypothesis: "H1".into(),
        endpoint: "held-r2".into(),
        context: "panel-v1".into(),
        protocol: "protocol-7".into(),
        independence_group: group.into(),
        source_head: HEAD.into(),
        source_committed: true,
        ci_head: Some(HEAD.into()),
        ci_passed: true,
        selection: SelectionMode::Prospective,
        outcome,
    }
}

fn pool_identity() -> PoolIdentity {
    PoolIdentity {
        hypothesis: "H1".into(),
        endpoint: "held-r2".into(),
        context: "panel-v1".into(),
        protocol: "protocol-7".into(),
        current_head: HEAD.into(),
    }
}

#[test]
fn unresolved_or_stale_source_and_ci_cannot_be_current_green() {
    assert_eq!(
        state_check(&source(), &ci(CiStatus::Pending, Some(HEAD), Some("run-1"))),
        CurrentState::CiPending
    );
    assert_eq!(
        state_check(
            &source(),
            &ci(CiStatus::Passed, Some(OLD_HEAD), Some("run-old"))
        ),
        CurrentState::StaleCi
    );
    assert_eq!(
        state_check(&source(), &ci(CiStatus::Passed, Some(HEAD), None)),
        CurrentState::CiUnknown
    );
    assert_eq!(
        state_check(
            &SourceEvidence {
                status: SourceStatus::Unresolved,
                head: Some(HEAD.into())
            },
            &ci(CiStatus::Passed, Some(HEAD), Some("run-2")),
        ),
        CurrentState::UnresolvedSource
    );
    assert_eq!(
        state_check(&source(), &ci(CiStatus::Passed, Some(HEAD), Some("run-3"))),
        CurrentState::CurrentGreen
    );
}

#[test]
fn repeat_warning_keeps_explicit_preregistered_replication_available() {
    let signature = signature();
    let prior = [PriorRun {
        run_id: "prior-1".into(),
        signature: signature.clone(),
    }];
    assert_eq!(
        do_not_repeat(&signature, &prior, None),
        DoNotRepeat::Warn {
            matching_run_ids: vec!["prior-1".into()]
        }
    );
    let prereg = ReplicationOverride {
        preregistration_id: "pre-registered-22".into(),
        reason: "Independent replication registered before outcomes".into(),
        registered_prospectively: true,
        independent_execution: true,
    };
    assert!(matches!(
        do_not_repeat(&signature, &prior, Some(&prereg)),
        DoNotRepeat::PermitPreregisteredReplication { .. }
    ));
}

#[test]
fn descriptive_pool_excludes_unresolved_correlated_and_incompatible_evidence() {
    let mut a = belief("a", "family-a", BeliefOutcome::Measured(0.5));
    let mut b = belief("b", "family-b", BeliefOutcome::Null);
    let c = belief("c", "family-c", BeliefOutcome::Inconclusive);
    let correlated = belief("d", "family-d", BeliefOutcome::Measured(0.9));
    let correlated_peer = belief("i", "family-d", BeliefOutcome::Measured(0.95));
    let mut stale = belief("e", "family-e", BeliefOutcome::Measured(0.8));
    let mut incompatible = belief("f", "family-f", BeliefOutcome::Measured(0.7));
    let mut adaptive = belief("g", "family-g", BeliefOutcome::Measured(0.6));
    let mut unresolved = belief("h", "family-h", BeliefOutcome::Measured(0.4));
    a.protocol = "protocol-7".into();
    b.protocol = "protocol-7".into();
    stale.source_head = OLD_HEAD.into();
    incompatible.endpoint = "train-loss".into();
    adaptive.selection = SelectionMode::Adaptive;
    unresolved.ci_passed = false;

    let summary = pool_descriptive(
        &pool_identity(),
        &[
            a,
            b,
            c,
            correlated,
            correlated_peer,
            stale,
            incompatible,
            adaptive,
            unresolved,
        ],
    );
    assert_eq!(
        summary.label,
        "descriptive only; not meta-analysis qualification"
    );
    assert_eq!(summary.measured_record_ids, vec!["a"]);
    assert_eq!(summary.null_record_ids, vec!["b"]);
    assert_eq!(summary.inconclusive_record_ids, vec!["c"]);
    assert_eq!(summary.mean, Some(0.5));
    assert_eq!(summary.denominator, 3);
    assert_eq!(summary.measured_denominator, 1);
    assert_eq!(summary.exclusions.len(), 6);
    assert!(summary.exclusions.contains(&Exclusion {
        record_id: "d".into(),
        reason: "correlated independence group".into()
    }));
    assert!(summary.exclusions.contains(&Exclusion {
        record_id: "i".into(),
        reason: "correlated independence group".into()
    }));
}

#[test]
fn brief_is_deterministic_and_enforces_the_byte_cap() {
    let entry = BriefEntry {
        id: "H1".into(),
        title: "Exploratory finding".into(),
        summary: "CI is pending".into(),
        source: source(),
        ci: ci(CiStatus::Pending, Some(HEAD), Some("run-4")),
        evidence_root: "https://evidence.invalid/I45/H1".into(),
    };
    let rendered = brief(&[entry.clone()]).unwrap();
    assert_eq!(rendered, brief(&[entry]).unwrap());
    assert!(rendered.len() <= STATE_BYTE_LIMIT);
    assert!(rendered.contains("CiPending"));
    assert!(rendered.contains("run-4"));
    assert!(rendered.contains("https://evidence.invalid/I45/H1"));

    let oversized = BriefEntry {
        id: "huge".into(),
        title: "oversized".into(),
        summary: "x".repeat(STATE_BYTE_LIMIT),
        source: SourceEvidence {
            status: SourceStatus::Unresolved,
            head: None,
        },
        ci: ci(CiStatus::Unknown, None, None),
        evidence_root: "evidence/H1".into(),
    };
    assert!(brief(&[oversized]).unwrap_err().contains("exceeding"));
}
