#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use jope_api::{
        BudgetTotal, BudgetUsed, Effort, ReasoningBody, RunPhase, RunState, TelemetryBody,
    };
    use serde_json::json;

    pub(crate) fn test_body(summary: &str) -> EventBody {
        EventBody::Reasoning(ReasoningBody {
            stage: "searching".into(),
            algorithm_id: None,
            summary: summary.into(),
            evidence: None,
        })
    }

    pub(crate) fn fixture() -> (tempfile::TempDir, Arc<EventLog>, RunSnapshot) {
        let directory = tempfile::tempdir().unwrap();
        let log = Arc::new(EventLog::new(Arc::new(
            Store::open(directory.path()).unwrap(),
        )));
        let snapshot = RunSnapshot {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            revision: DecimalU64(1),
            state: RunState::Paused,
            phase: RunPhase::Searching,
            pending: None,
            effort: Effort::Medium,
            budget_used: BudgetUsed {
                evaluations: DecimalU64(0),
                wall_seconds: "0".into(),
            },
            budget_total: BudgetTotal {
                evaluations: DecimalU64(128),
                wall_seconds: DecimalU64(60),
            },
            best: None,
            event_watermark: DecimalU64(0),
            warnings: vec![],
            completion_reason: None,
            evaluation_receipt: None,
        };
        log.commit(
            RunUpdate {
                expected_revision: None,
                snapshot: snapshot.clone(),
                checkpoint: json!({"round": 0}),
                blobs: vec![],
            },
            vec![],
            |_| Ok(()),
        )
        .unwrap();
        (directory, log, snapshot)
    }

    pub(crate) fn append(
        log: &EventLog,
        snapshot: &RunSnapshot,
        bodies: Vec<EventBody>,
    ) -> RunSnapshot {
        let mut next = snapshot.clone();
        next.revision.0 += 1;
        log.commit(
            RunUpdate {
                expected_revision: Some(snapshot.revision.0),
                snapshot: next,
                checkpoint: json!({"round": snapshot.revision.0}),
                blobs: vec![],
            },
            bodies,
            |_| Ok(()),
        )
        .unwrap()
        .snapshot
    }

    include!("tests_cases.rs");
}
