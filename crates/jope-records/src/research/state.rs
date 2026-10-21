//! Deterministic, bounded research-state helpers.
//!
//! This module only interprets caller-supplied evidence. It does not rewrite
//! coordination history, protocol documents, or canonical generated files.

use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const STATE_BYTE_LIMIT: usize = 8_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceStatus {
    Committed,
    Unresolved,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiStatus {
    Passed,
    Failed,
    Pending,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceEvidence {
    pub status: SourceStatus,
    pub head: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CiEvidence {
    pub status: CiStatus,
    pub head: Option<String>,
    pub run_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurrentState {
    CurrentGreen,
    UnresolvedSource,
    RejectedSource,
    SourceHeadUnknown,
    CiUnknown,
    CiPending,
    CiFailed,
    CiHeadUnknown,
    StaleCi,
}

/// Report current-green only for committed source with a passing CI result at
/// exactly the same, explicitly supplied full source head.
pub fn state_check(source: &SourceEvidence, ci: &CiEvidence) -> CurrentState {
    match source.status {
        SourceStatus::Unresolved => return CurrentState::UnresolvedSource,
        SourceStatus::Rejected => return CurrentState::RejectedSource,
        SourceStatus::Committed => {}
    }
    let Some(source_head) = source.head.as_deref().filter(|head| is_full_sha(head)) else {
        return CurrentState::SourceHeadUnknown;
    };
    match ci.status {
        CiStatus::Unknown => CurrentState::CiUnknown,
        CiStatus::Pending => CurrentState::CiPending,
        CiStatus::Failed => CurrentState::CiFailed,
        CiStatus::Passed => {
            if ci
                .run_id
                .as_deref()
                .is_none_or(|run_id| run_id.trim().is_empty())
            {
                return CurrentState::CiUnknown;
            }
            let Some(ci_head) = ci.head.as_deref().filter(|head| is_full_sha(head)) else {
                return CurrentState::CiHeadUnknown;
            };
            if ci_head == source_head {
                CurrentState::CurrentGreen
            } else {
                CurrentState::StaleCi
            }
        }
    }
}

fn is_full_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RepeatSignature {
    pub hypothesis: String,
    pub endpoint: String,
    pub context: String,
    pub support: String,
    pub cost: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PriorRun {
    pub run_id: String,
    pub signature: RepeatSignature,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicationOverride {
    pub preregistration_id: String,
    pub reason: String,
    /// The registration was recorded before outcome data were observed.
    pub registered_prospectively: bool,
    /// The planned replication has an independent execution identity.
    pub independent_execution: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DoNotRepeat {
    Novel,
    Warn {
        matching_run_ids: Vec<String>,
    },
    PermitPreregisteredReplication {
        matching_run_ids: Vec<String>,
        preregistration_id: String,
        reason: String,
    },
}

/// Warn on an exact repeat signature, while retaining a prospective and
/// independently executed preregistered replication when its reason is given.
pub fn do_not_repeat(
    signature: &RepeatSignature,
    history: &[PriorRun],
    replication: Option<&ReplicationOverride>,
) -> DoNotRepeat {
    let mut matching_run_ids: Vec<_> = history
        .iter()
        .filter(|prior| prior.signature == *signature)
        .map(|prior| prior.run_id.clone())
        .collect();
    matching_run_ids.sort();
    matching_run_ids.dedup();
    if matching_run_ids.is_empty() {
        return DoNotRepeat::Novel;
    }
    if let Some(override_record) = replication.filter(|candidate| {
        candidate.registered_prospectively
            && candidate.independent_execution
            && !candidate.preregistration_id.trim().is_empty()
            && !candidate.reason.trim().is_empty()
    }) {
        return DoNotRepeat::PermitPreregisteredReplication {
            matching_run_ids,
            preregistration_id: override_record.preregistration_id.clone(),
            reason: override_record.reason.clone(),
        };
    }
    DoNotRepeat::Warn { matching_run_ids }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BeliefOutcome {
    Measured(f64),
    Null,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Prospective,
    Adaptive,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BeliefRecord {
    pub record_id: String,
    pub hypothesis: String,
    pub endpoint: String,
    pub context: String,
    pub protocol: String,
    pub independence_group: String,
    pub source_head: String,
    pub source_committed: bool,
    pub ci_head: Option<String>,
    pub ci_passed: bool,
    pub selection: SelectionMode,
    pub outcome: BeliefOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolIdentity {
    pub hypothesis: String,
    pub endpoint: String,
    pub context: String,
    pub protocol: String,
    pub current_head: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exclusion {
    pub record_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DescriptivePool {
    /// This output is descriptive only and never qualifies a meta-analysis.
    pub label: &'static str,
    pub measured_record_ids: Vec<String>,
    pub null_record_ids: Vec<String>,
    pub inconclusive_record_ids: Vec<String>,
    pub mean: Option<f64>,
    /// All valid independent compatible records, including null and inconclusive outcomes.
    pub denominator: usize,
    /// Measured outcomes that contribute numeric values to `mean`.
    pub measured_denominator: usize,
    pub exclusions: Vec<Exclusion>,
}

/// Pool only exact-head, passing, prospective, compatible evidence whose
/// independence groups are unique. All exclusions and non-measured outcomes
/// stay visible in the result.
pub fn pool_descriptive(identity: &PoolIdentity, records: &[BeliefRecord]) -> DescriptivePool {
    let mut ordered: Vec<_> = records.iter().collect();
    ordered.sort_by(|left, right| left.record_id.cmp(&right.record_id));
    let mut reasons = BTreeMap::<String, String>::new();
    let mut candidate_groups = BTreeMap::<String, Vec<String>>::new();
    let mut record_id_counts = BTreeMap::<String, usize>::new();
    for record in &ordered {
        *record_id_counts
            .entry(record.record_id.clone())
            .or_default() += 1;
    }

    for record in &ordered {
        let reason = if record.record_id.trim().is_empty() {
            Some("missing record identity")
        } else if record_id_counts[&record.record_id] > 1 {
            Some("duplicate record identity")
        } else if !record.source_committed {
            Some("source is unresolved")
        } else if !is_full_sha(&record.source_head) {
            Some("source head is unknown")
        } else if record.source_head != identity.current_head {
            Some("source head is stale")
        } else if !record.ci_passed {
            Some("CI is unresolved or failed")
        } else if record.ci_head.as_deref() != Some(record.source_head.as_str()) {
            Some("CI is not bound to the source head")
        } else if record.hypothesis != identity.hypothesis
            || record.endpoint != identity.endpoint
            || record.context != identity.context
            || record.protocol != identity.protocol
        {
            Some("incompatible hypothesis, endpoint, context, or protocol")
        } else if record.selection != SelectionMode::Prospective {
            Some("adaptive selection")
        } else if record.independence_group.trim().is_empty() {
            Some("missing independence group")
        } else if matches!(record.outcome, BeliefOutcome::Measured(value) if !value.is_finite()) {
            Some("non-finite measured outcome")
        } else {
            None
        };
        if let Some(reason) = reason {
            reasons.insert(record.record_id.clone(), reason.to_string());
        } else {
            candidate_groups
                .entry(record.independence_group.clone())
                .or_default()
                .push(record.record_id.clone());
        }
    }

    for record_ids in candidate_groups.values().filter(|ids| ids.len() > 1) {
        for record_id in record_ids {
            reasons.insert(
                record_id.clone(),
                "correlated independence group".to_string(),
            );
        }
    }

    let mut measured_record_ids = Vec::new();
    let mut null_record_ids = Vec::new();
    let mut inconclusive_record_ids = Vec::new();
    let mut values = Vec::new();
    for record in ordered {
        if reasons.contains_key(&record.record_id) {
            continue;
        }
        match record.outcome {
            BeliefOutcome::Measured(value) => {
                measured_record_ids.push(record.record_id.clone());
                values.push(value);
            }
            BeliefOutcome::Null => null_record_ids.push(record.record_id.clone()),
            BeliefOutcome::Inconclusive => inconclusive_record_ids.push(record.record_id.clone()),
        }
    }
    let mut exclusions: Vec<_> = reasons
        .into_iter()
        .map(|(record_id, reason)| Exclusion { record_id, reason })
        .collect();
    exclusions.sort_by(|left, right| left.record_id.cmp(&right.record_id));
    let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
    let denominator =
        measured_record_ids.len() + null_record_ids.len() + inconclusive_record_ids.len();
    DescriptivePool {
        label: "descriptive only; not meta-analysis qualification",
        measured_record_ids,
        null_record_ids,
        inconclusive_record_ids,
        mean,
        denominator,
        measured_denominator: values.len(),
        exclusions,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BriefEntry {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub source: SourceEvidence,
    pub ci: CiEvidence,
    pub evidence_root: String,
}

/// Render one deterministic entry point, bounded by the 8,000-byte protocol cap.
/// Does not write or relocate any source history.
pub fn brief(entries: &[BriefEntry]) -> Result<String, String> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| left.id.cmp(&right.id));
    if sorted.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err("duplicate state entry id".to_string());
    }
    let mut output = String::from(
        "# Research state\n\nGenerated as a compact view of supplied evidence. Source, CI, runtime, science, and serving states remain distinct.\n\n",
    );
    for entry in sorted {
        writeln!(
            &mut output,
            "## {} — {}",
            markdown_cell(&entry.id),
            markdown_cell(&entry.title)
        )
        .expect("writing to String cannot fail");
        writeln!(&mut output, "\n{}\n", markdown_cell(&entry.summary))
            .expect("writing to String cannot fail");
        writeln!(
            &mut output,
            "- State: `{:?}`\n- Source: `{}`\n- CI: `{}`\n- CI run: `{}`\n- Evidence: <{}>\n",
            state_check(&entry.source, &entry.ci),
            code_cell(entry.source.head.as_deref().unwrap_or("unknown")),
            code_cell(entry.ci.head.as_deref().unwrap_or("unknown")),
            code_cell(entry.ci.run_id.as_deref().unwrap_or("unknown")),
            safe_link(&entry.evidence_root)
        )
        .expect("writing to String cannot fail");
    }
    if output.len() > STATE_BYTE_LIMIT {
        return Err(format!(
            "rendered state is {} bytes, exceeding the {STATE_BYTE_LIMIT}-byte cap",
            output.len()
        ));
    }
    Ok(output)
}

fn markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace(['\r', '\n'], " ")
}

fn safe_link(value: &str) -> String {
    if value.contains(['<', '>', '\n', '\r']) {
        "invalid evidence link".to_string()
    } else {
        value.to_string()
    }
}

fn code_cell(value: &str) -> String {
    markdown_cell(value).replace('`', "\\`")
}

#[cfg(test)]
mod tests {
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
}
