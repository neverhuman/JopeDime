use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SupervisorHandle {
    pub supervisor: String,
    pub identity: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "cause", rename_all = "snake_case", deny_unknown_fields)]
pub enum DrainCause {
    RankFailed { global_rank: u32, detail: String },
    ControlLost { global_rank: u32 },
    GrantExpired { global_rank: u32 },
    DeadlineElapsed,
    WallBudgetElapsed { global_rank: u32 },
    Cancelled,
    CoordinatorRestart,
}
/// When [`group_window_closed`] first closes the window (Unix ms).
pub(crate) fn group_window_end(contract: &BoundedRunContract) -> u64 {
    contract
        .participants
        .iter()
        .map(|p| p.grant.body.expires_unix_ms)
        .fold(contract.original_deadline_unix_ms, u64::min)
}
/// The one execution window of the synchronous group: it closes at the original deadline
/// or at the first participant grant expiry, whichever comes first. Supervision, restart
/// recovery, dispatch, admission, delivery, acknowledgement and every new commit use this
/// single gate; only finishing an already-journaled commit or drain may happen later.
pub fn group_window_closed(contract: &BoundedRunContract, now: u64) -> Option<DrainCause> {
    if now >= contract.original_deadline_unix_ms {
        return Some(DrainCause::DeadlineElapsed);
    }
    contract
        .participants
        .iter()
        .find(|p| now >= p.grant.body.expires_unix_ms)
        .map(|p| DrainCause::GrantExpired {
            global_rank: p.global_rank,
        })
}
/// Wall consumption starts at the first durable dispatch intent, not at recovery.
pub(crate) fn journal_window_closed(
    contract: &BoundedRunContract,
    records: &[(JournalRecord, String)],
    now: u64,
) -> Option<DrainCause> {
    if let Some(cause) = group_window_closed(contract, now) {
        return Some(cause);
    }
    records.iter().find_map(|(record, _)| {
        let BoundedControllerEvent::DispatchIntent { global_rank, .. } = record.event else {
            return None;
        };
        let participant = contract.participant(global_rank)?;
        (now.saturating_sub(record.recorded_unix_ms) >= participant.ceilings.wall_ms)
            .then_some(DrainCause::WallBudgetElapsed { global_rank })
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundedControllerDisposition {
    Committed,
    /// The rank whose failure caused the drain.
    Failed,
    /// Stopped by the whole-group drain while running (or already stopped).
    Drained,
    /// No work was ever started for this rank.
    NeverStarted,
    /// An authenticated result was retained but never acknowledged; not committed.
    CompletedUnacknowledged,
    /// An authenticated result was acknowledged but the group drained before commit.
    CompletedUncommitted,
    /// The group was refused before any rank started.
    Refused,
}

/// Compatibility alias for the domain-specific `BoundedControllerDisposition` type.
pub type Disposition = BoundedControllerDisposition;
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundedControllerOutcome {
    Committed,
    Drained,
    Cancelled,
    Refused,
}

/// Compatibility alias for the bounded-controller terminal result.
pub type Outcome = BoundedControllerOutcome;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum BoundedControllerEvent {
    Planned {
        original_deadline_unix_ms: u64,
        committed_updates_at_start: u64,
        planned_updates: u64,
    },
    PreflightAdmitted {
        receipt: PreflightReceipt,
    },
    PreflightRefused {
        census_sha256: String,
        reasons: Vec<RefusalReason>,
    },
    GroupAdmitted {
        admission_sha256: String,
        admission: GroupAdmission,
    },
    /// Every participant was sent the persisted admission; ranks may now start work.
    AdmissionDelivered {
        admission_sha256: String,
    },
    GroupRefused {
        refusal: GroupRefusal,
    },
    DispatchIntent {
        global_rank: u32,
        dispatch_id: String,
    },
    Launched {
        global_rank: u32,
        dispatch_id: String,
        handle: SupervisorHandle,
    },
    ResultAcknowledged {
        global_rank: u32,
        delivery_id: u64,
        result_sha256: String,
        envelope_sha256: String,
        updates: u64,
    },
    GroupCommitted {
        committed_updates_total: u64,
        /// `(global_rank, result_sha256)` in rank order (JSON map keys cannot be integers
        /// inside a tagged record).
        results: Vec<(u32, String)>,
    },
    CancelRequested {
        request_sha256: String,
    },
    CoordinatorRestarted {
        boot_id: String,
        pid: u32,
    },
    DrainStarted {
        cause: DrainCause,
    },
    BoundedControllerDisposition {
        global_rank: u32,
        disposition: BoundedControllerDisposition,
        detail: String,
    },
    Terminal {
        outcome: BoundedControllerOutcome,
        committed_updates_total: u64,
    },
}

/// Compatibility alias for the domain-specific `BoundedControllerEvent` type.
pub type Event = BoundedControllerEvent;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JournalRecord {
    pub schema: String,
    pub sequence: u64,
    pub previous_sha256: Option<String>,
    pub plan_sha256: String,
    pub recorded_unix_ms: u64,
    pub event: BoundedControllerEvent,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CancelRequest {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub reason: String,
    pub requested_unix_ms: u64,
}

/// The seven externally visible rank states.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RankState {
    Scheduled,
    Admitted,
    Running,
    Committed,
    Failed,
    Drained,
    Unknown,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RankStatus {
    pub global_rank: u32,
    pub host: String,
    pub device_uuid: Option<String>,
    pub state: RankState,
    pub disposition: Option<BoundedControllerDisposition>,
    pub detail: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StatusReport {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub original_deadline_unix_ms: u64,
    pub journal_records: u64,
    pub journal_tip_sha256: Option<String>,
    pub coordinator_live: bool,
    pub cancel_requested: bool,
    pub outcome: Option<BoundedControllerOutcome>,
    pub committed_updates_at_start: u64,
    pub committed_updates_total: u64,
    pub ranks: Vec<RankStatus>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifyReport {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub journal_records: u64,
    pub journal_tip_sha256: Option<String>,
    pub outcome: Option<BoundedControllerOutcome>,
    pub committed_updates_total: u64,
    pub results_verified: u64,
    pub numerical_work_started: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanReport {
    pub plan_sha256: String,
    pub created: bool,
    pub review: serde_json::Value,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CancelReport {
    pub plan_sha256: String,
    pub request: CancelRequest,
    pub already_requested: bool,
    pub journaled: bool,
    pub delivered_to_live_coordinator: bool,
    pub drain_pending_reconcile: bool,
    pub outcome: Option<BoundedControllerOutcome>,
}
