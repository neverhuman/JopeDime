use super::*;

use super::*;

const HEAD_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HEAD_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn record(id: &str, group: &str) -> BeliefRecord {
    BeliefRecord {
        record_id: id.into(),
        hypothesis: "H1".into(),
        endpoint: "held-r2".into(),
        context: "panel-v1".into(),
        protocol: "protocol-7".into(),
        independence_group: group.into(),
        source_head: HEAD_A.into(),
        source_committed: true,
        ci_head: Some(HEAD_A.into()),
        ci_passed: true,
        selection: SelectionMode::Prospective,
        outcome: BeliefOutcome::Measured(0.7),
    }
}

#[test]
fn unresolved_or_stale_source_and_ci_never_render_current_green() {
    let source = SourceEvidence {
        status: SourceStatus::Committed,
        head: Some(HEAD_A.into()),
    };
    assert_eq!(
        state_check(
            &source,
            &CiEvidence {
                status: CiStatus::Passed,
                head: Some(HEAD_B.into()),
                run_id: Some("run1".into())
            }
        ),
        CurrentState::StaleCi
    );
    for status in [CiStatus::Unknown, CiStatus::Pending, CiStatus::Failed] {
        assert_ne!(
            state_check(
                &source,
                &CiEvidence {
                    status,
                    head: Some(HEAD_A.into()),
                    run_id: None
                }
            ),
            CurrentState::CurrentGreen
        );
    }
    assert_eq!(
        state_check(
            &SourceEvidence {
                status: SourceStatus::Unresolved,
                head: Some(HEAD_A.into())
            },
            &CiEvidence {
                status: CiStatus::Passed,
                head: Some(HEAD_A.into()),
                run_id: None
            }
        ),
        CurrentState::UnresolvedSource
    );
    assert_eq!(
        state_check(
            &source,
            &CiEvidence {
                status: CiStatus::Passed,
                head: None,
                run_id: Some("run-4".into())
            }
        ),
        CurrentState::CiHeadUnknown
    );
    assert_eq!(
        state_check(
            &source,
            &CiEvidence {
                status: CiStatus::Passed,
                head: Some(HEAD_A.into()),
                run_id: None
            }
        ),
        CurrentState::CiUnknown
    );
    assert_eq!(
        state_check(
            &source,
            &CiEvidence {
                status: CiStatus::Passed,
                head: Some(HEAD_A.into()),
                run_id: Some("   ".into())
            }
        ),
        CurrentState::CiUnknown
    );
    assert_eq!(
        state_check(
            &source,
            &CiEvidence {
                status: CiStatus::Passed,
                head: Some(HEAD_A.into()),
                run_id: Some("run-3".into())
            }
        ),
        CurrentState::CurrentGreen
    );
}

#[test]
fn exact_repeat_warns_but_explicit_preregistered_replication_is_preserved() {
    let signature = RepeatSignature {
        hypothesis: "H1".into(),
        endpoint: "held-r2".into(),
        context: "panel-v1".into(),
        support: "dataset-set-3".into(),
        cost: "cpu-cap-2h".into(),
    };
    let history = [PriorRun {
        run_id: "run-old".into(),
        signature: signature.clone(),
    }];
    assert_eq!(
        do_not_repeat(&signature, &history, None),
        DoNotRepeat::Warn {
            matching_run_ids: vec!["run-old".into()]
        }
    );
    let override_record = ReplicationOverride {
        preregistration_id: "prereg-22".into(),
        reason: "Independent replication planned before outcomes".into(),
        registered_prospectively: true,
        independent_execution: true,
    };
    assert_eq!(
        do_not_repeat(&signature, &history, Some(&override_record)),
        DoNotRepeat::PermitPreregisteredReplication {
            matching_run_ids: vec!["run-old".into()],
            preregistration_id: "prereg-22".into(),
            reason: "Independent replication planned before outcomes".into(),
        }
    );
    let invalid_override = ReplicationOverride {
        registered_prospectively: false,
        ..override_record
    };
    assert!(matches!(
        do_not_repeat(&signature, &history, Some(&invalid_override)),
        DoNotRepeat::Warn { .. }
    ));
}

#[test]
fn pooled_descriptive_summary_excludes_unresolved_correlated_and_incompatible_records() {
    let identity = PoolIdentity {
        hypothesis: "H1".into(),
        endpoint: "held-r2".into(),
        context: "panel-v1".into(),
        protocol: "protocol-7".into(),
        current_head: HEAD_A.into(),
    };
    let mut first = record("a", "group-a");
    first.outcome = BeliefOutcome::Measured(0.6);
    let mut second = record("b", "group-b");
    second.outcome = BeliefOutcome::Null;
    let mut third = record("c", "group-c");
    third.outcome = BeliefOutcome::Inconclusive;
    let mut correlated = record("d", "group-d");
    let mut correlated_peer = record("i", "group-d");
    correlated_peer.outcome = BeliefOutcome::Measured(0.95);
    let mut stale = record("e", "group-e");
    stale.source_head = HEAD_B.into();
    let mut ci_pending = record("f", "group-f");
    ci_pending.ci_passed = false;
    let mut incompatible = record("g", "group-g");
    incompatible.endpoint = "train-loss".into();
    let mut adaptive = record("h", "group-h");
    adaptive.selection = SelectionMode::Adaptive;
    correlated.outcome = BeliefOutcome::Measured(0.99);
    let summary = pool_descriptive(
        &identity,
        &[
            incompatible,
            ci_pending,
            second,
            adaptive,
            stale,
            first,
            correlated,
            correlated_peer,
            third,
        ],
    );
    assert_eq!(
        summary.label,
        "descriptive only; not meta-analysis qualification"
    );
    assert_eq!(summary.measured_record_ids, vec!["a"]);
    assert_eq!(summary.null_record_ids, vec!["b"]);
    assert_eq!(summary.inconclusive_record_ids, vec!["c"]);
    assert_eq!(summary.mean, Some(0.6));
    assert_eq!(summary.denominator, 3);
    assert_eq!(summary.measured_denominator, 1);
    let reasons: BTreeMap<_, _> = summary
        .exclusions
        .iter()
        .map(|item| (item.record_id.as_str(), item.reason.as_str()))
        .collect();
    assert_eq!(reasons["d"], "correlated independence group");
    assert_eq!(reasons["i"], "correlated independence group");
    assert_eq!(reasons["e"], "source head is stale");
    assert_eq!(reasons["f"], "CI is unresolved or failed");
    assert_eq!(
        reasons["g"],
        "incompatible hypothesis, endpoint, context, or protocol"
    );
    assert_eq!(reasons["h"], "adaptive selection");
}

#[test]
fn brief_is_deterministic_linked_and_capped_without_truncation() {
    let entry = BriefEntry {
        id: "H1".into(),
        title: "Test hypothesis".into(),
        summary: "CI is pending".into(),
        source: SourceEvidence {
            status: SourceStatus::Committed,
            head: Some(HEAD_A.into()),
        },
        ci: CiEvidence {
            status: CiStatus::Pending,
            head: Some(HEAD_A.into()),
            run_id: Some("run-1".into()),
        },
        evidence_root: "https://evidence.invalid/I45/H1".into(),
    };
    let rendered = brief(std::slice::from_ref(&entry)).unwrap();
    assert_eq!(rendered, brief(&[entry]).unwrap());
    assert!(rendered.len() <= STATE_BYTE_LIMIT);
    assert!(rendered.contains("CiPending"));
    assert!(rendered.contains("https://evidence.invalid/I45/H1"));
    let huge = BriefEntry {
        summary: "x".repeat(STATE_BYTE_LIMIT),
        ..brief_entry()
    };
    assert!(brief(&[huge]).unwrap_err().contains("exceeding"));
    assert!(matches!(
        pool_descriptive(
            &PoolIdentity {
                hypothesis: "".into(),
                endpoint: "".into(),
                context: "".into(),
                protocol: "".into(),
                current_head: HEAD_A.into()
            },
            &[]
        )
        .label,
        "descriptive only; not meta-analysis qualification"
    ));
}

fn brief_entry() -> BriefEntry {
    BriefEntry {
        id: "H2".into(),
        title: "large".into(),
        summary: String::new(),
        source: SourceEvidence {
            status: SourceStatus::Unresolved,
            head: None,
        },
        ci: CiEvidence {
            status: CiStatus::Unknown,
            head: None,
            run_id: None,
        },
        evidence_root: "evidence/I45/H2".into(),
    }
}
