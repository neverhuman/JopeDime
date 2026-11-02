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
mod tests;
